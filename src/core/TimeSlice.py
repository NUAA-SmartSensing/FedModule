import ast
import importlib
import inspect
import os
import sys
import time
from abc import abstractmethod
from collections import ChainMap
from multiprocessing import Process
from types import MappingProxyType as readonlydict

from core.Runtime import Mode, CLIENT_STATUS
from utils import ModuleFindTool
from utils.GlobalVarGetter import GlobalVarGetter
from utils.ProcessManager import MessageQueueFactory


class TimeSlice(Mode):
    """
    [experimental]
    TimeSlice is a class that uses time slices to simulate client work and server work.
    """

    def __init__(self):
        pass

    @abstractmethod
    def run(self):
        pass

    def start(self):
        pass

    def join(self):
        """
        Join the client timeslice.
        """
        pass


class ModuleUseCollector(ast.NodeVisitor):
    def __init__(self, modulename, package='', attr='', modified_module={}):
        self.modulename = modulename
        self.modified_module = modified_module
        self.attr = attr
        # used to resolve from ... import ... references
        self.package = package
        self.modulepackage, _, self.modulestem = modulename.rpartition('.')
        # track scope namespaces, with a mapping of imported names (bound name to original)
        # If a name references None it is used for a different purpose in that scope
        # and so masks a name in the global namespace.
        self.scopes = ChainMap()
        self.used_at = []  # list of (name, alias, line) entries
        self.status = 0
        self.current_klass = None
        self.functionsCall = []
        self.recorded_module = []

    def visit_FunctionDef(self, node):
        self.scopes = self.scopes.new_child()
        self.functionsCall.append(node.name)
        self.generic_visit(node)
        self.functionsCall.pop()
        self.scopes = self.scopes.parents

    def visit_Lambda(self, node):
        # lambdas are just functions, albeit with no statements
        self.visit_Function(node)

    def visit_ClassDef(self, node):
        # class scope is a special local scope that is re-purposed to form
        # the class attributes. By using a read-only dict proxy here this code
        # we can expect an exception when a class body contains an import
        # statement or uses names that'd mask an imported name.
        self.scopes = self.scopes.new_child(readonlydict({}))
        self.functionsCall = []
        self.current_klass = node.name
        self.generic_visit(node)
        self.current_klass = None
        self.functionsCall = []
        self.scopes = self.scopes.parents

    def visit_Import(self, node):
        # if the module import another modified module, we need to record it
        # and Repoint it to the modified module
        self.scopes.update({
            a.asname or a.name: a.name
            for a in node.names
            if a.name == self.modulename
        })
        for a in node.names:
            if a.name in self.modified_module:
                self.recorded_module.append((a.asname or a.name, None))

    def visit_ImportFrom(self, node):
        # resolve relative imports; from . import <name>, from ..<name> import <name>
        source = node.module  # can be None
        if node.level:
            package = self.package
            if node.level > 1:
                # go up levels as needed
                package = '.'.join(self.package.split('.')[:-(node.level - 1)])
            source = f'{package}.{source}' if source else package
        if self.modulename == source:
            self.status = 1
            # names imported from our target module
            self.scopes.update({
                a.asname or a.name: f'{self.modulename}.{a.name}'
                for a in node.names
            })
        elif self.modulepackage and self.modulepackage == source:
            # from package import module import, where package.module is what we want
            self.scopes.update({
                a.asname or a.name: self.modulename
                for a in node.names
                if a.name == self.modulestem
            })
        if source in self.modified_module:
            self.recorded_module.append((source, [a.asname or a.name for a in node.names]))
        else:
            for a in node.names:
                if f'{source}.{a.name}' in node.names:
                    self.recorded_module.append((f'{source}.{a.name}', None))

    def visit_Name(self, node):
        if not isinstance(node.ctx, ast.Load):
            # store or del operation, means the name is masked in the current scope
            try:
                self.scopes[node.id] = None
            except TypeError:
                # class scope, which we made read-only. These names can't mask
                # anything so just ignore these.
                pass
            return
        if self.status:
            # find scope this name was defined in, starting at the current scope
            imported_name = self.scopes.get(node.id)
            if imported_name is None:
                return
            self.used_at.append((imported_name, node.id, node.lineno, ".".join(self.functionsCall), self.current_klass))

    def visit_Attribute(self, node):
        if not self.status and isinstance(node.value, ast.Name):
            imported_name = self.scopes.get(node.value.id)
            if imported_name is None:
                return
            elif node.attr == self.attr:
                self.used_at.append(
                    (imported_name, node.attr, node.lineno, ".".join(self.functionsCall), self.current_klass))


class MethodUseCollector(ast.NodeVisitor):
    def __init__(self, klass, methods):
        self.klass = klass
        self.methods = methods
        self.methodCall = []
        self.used_at = set()
        self.function_overloaded = []
        self.current_klass = None
        self.is_subclass = False
        self.is_self = False
        self.class_src = {}

    def visit_FunctionDef(self, node):
        self.methodCall.append(node.name)
        if self.is_subclass and node.name in self.methods:
            self.function_overloaded.append(node.name)
        elif self.is_self and node.name in self.methods:
            return
        self.generic_visit(node)
        self.methodCall.pop()

    def visit_Lambda(self, node):
        # lambdas are just functions, albeit with no statements
        self.visit_Function(node)

    def visit_ClassDef(self, node):
        self.methodCall = [node.name]
        # its bases may be a function, that situation we can't solve
        # so we just ignore it
        # we just solve two situations:
        # 1. class A(B): pass
        # 2. class A(B.C): pass
        if node.name == self.klass:
            self.is_self = True
        else:
            if node.bases:
                for child_node in node.bases:
                    if isinstance(child_node, ast.Name) and child_node.id == self.klass:
                        self.is_subclass = True
                        break
                    elif isinstance(child_node, ast.Attribute) and not isinstance(child_node.value,
                                                                                  ast.Attribute) and child_node.attr == self.klass:
                        self.is_subclass = True
                        break
                    else:
                        raise Exception(f'Class {self.klass}\'s bases are too deep. We can\'t solve it.\n'
                                        f'Please inherit the class directly: class A(B) or class A(B.C)')

        if not self.is_self and not self.is_subclass:
            raise Exception(f'Class {self.klass} not found in range.')
        self.methodCall = []
        self.current_klass = node.name
        self.generic_visit(node)
        self.current_klass = None
        self.methodCall = []
        self.is_self = False
        self.is_subclass = False
        self.methodCall = []

    def visit_Attribute(self, node):
        if isinstance(node.value, ast.Name) and node.attr in self.methods:
            if self.is_self and node.value.id == 'self':
                self.used_at.add((node.lineno, ".".join(self.methodCall), self.current_klass))
            elif self.is_subclass:
                # A.call -- Attribute(
                #                 value=Name(
                #                   id='A',
                #                   ctx=Load()),
                #                 attr='call',
                #                 ctx=Load())
                self.used_at.add((node.lineno, ".".join(self.methodCall), self.current_klass))
        elif self.is_subclass and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
            if node.attr in self.methods and node.value.func.id == 'super':
                # super().call --Attribute(
                #               value=Call(
                #                 func=Name(
                #                   id='super',
                #                   ctx=Load()),
                #                 args=[],
                #                 keywords=[]),
                #               attr='call',
                #               ctx=Load())
                self.used_at.add((node.lineno, ".".join(self.methodCall), self.current_klass))


def modify_and_import(module_name, package, modification_func, *args, **kwargs):
    spec = importlib.util.find_spec(module_name, package)
    source = spec.loader.get_source(module_name)
    new_source = modification_func(source, *args, **kwargs)
    module = importlib.util.module_from_spec(spec)
    codeobj = compile(new_source, module.__spec__.origin, 'exec')
    exec(codeobj, module.__dict__)
    sys.modules[module_name] = module
    return module


def replace(source, *args, **kwargs):
    # we cant resolve sentence like "a = call()" or "return call()"
    # such sentence is not complicated with yield or yield from
    modified_line = args[0]
    source = source.split("\n")
    yield_line = modified_line["yield"]
    yield_from_line = modified_line["yield_from"]
    for line in yield_line:
        line = line - 1
        inserted_index = len(source[line]) - len(source[line].lstrip())
        source[line] = source[line][:inserted_index] + f'yield ' + source[line][inserted_index:]
    for line in yield_from_line:
        line = line - 1
        inserted_index = len(source[line]) - len(source[line].lstrip())
        source[line] = source[line][:inserted_index] + f'yield from ' + source[line][inserted_index:]
    return "\n".join(source)


class TimeSliceRunner(Process):
    """
    [experimental]
    RunnerProcess is the process of the timeslice runner.
    """

    def __init__(self, init_client_event, create_client_event, join_event, stop_event_list, selected_event_list,
                 config):
        super().__init__()
        # we need client manager delegate the power to create clients to it.
        self.client_staleness_list = None
        self.index_list = None
        self.client_num = None
        self.client_class = None
        self.client_status = None
        self.message_queue = None
        self.client_list = []
        self.init_client_event = init_client_event
        self.create_client_event = create_client_event
        self.join_event = join_event
        self.stop_event_list = stop_event_list
        self.selected_event_list = selected_event_list
        self.config = config

    @staticmethod
    def func_decorator(func):
        def wrapper(*args, **kwargs):
            def yield_wrapper(inner_func, *args, **kwargs):
                yield from func(*args, **kwargs)

            for name in dir(func):
                attr = getattr(func, name)
                if callable(attr):
                    setattr(func, name, yield_wrapper(attr, *args, **kwargs))

        return wrapper

    def run(self):
        # process should init itself before running
        self.init()
        # create clients and start clients
        if self.init_client_event.is_set():
            self.create_and_start_all_clients()

        # run clients
        self.client_simulate()

        self.join_event.set()

    def client_simulate(self):
        runner = [client.run() for client in self.client_list]
        timeline = [0 for _ in self.client_list]
        with open(os.path.join("../results/", GlobalVarGetter.get()["global"]["experiment"], "client_status.txt")) as file:
            file.write(str(timeline))
            while not self.stop_event_list.is_set():
                for i, client in enumerate(self.client_list):
                    if self.selected_event_list[i].is_set():
                        if timeline[i] == 0:
                            sec = 0
                            try:
                                sec = next(runner[i])
                            except StopIteration:
                                self.client_status[i] = CLIENT_STATUS["stopped"]
                            timeline[i] += sec
                        else:
                            timeline[i] -= 1

    def create_and_start_all_clients(self):
        for i in range(self.client_num):
            self.client_list.append(
                self.client_class(i, self.stop_event_list[i], self.selected_event_list[i],
                                  self.client_staleness_list[i],
                                  self.index_list[i], self.config['client_config'], self.config['client_dev'][i])
            )
        self.client_status = [CLIENT_STATUS["active"] for _ in range(len(self.config["client_num"]))]
        for client in self.client_list:
            client.run = self.func_decorator(client.run)

    def create_client(self):
        client_delay = self.message_queue.get_from_downlink(-1, "client_delay")
        dev = self.message_queue.get_from_downlink(-1, "dev")
        client = self.client_class(len(self.client_list), self.stop_event_list[len(self.client_list)],
                                   self.selected_event_list[len(self.client_list)], client_delay,
                                   self.index_list[len(self.client_list)], self.config['client_config'], dev)
        client.run = self.func_decorator(client.run)
        self.client_list.append(client)
        self.client_status.append(CLIENT_STATUS["active"])

    def init(self):
        self.message_queue = MessageQueueFactory.create_message_queue()
        self.client_class = ModuleFindTool.find_class_by_path(self.config["client_config"]["path"])
        self.client_num = self.config["client_num"]
        self.index_list = self.config["index_list"]
        self.client_staleness_list = self.config["client_staleness_list"]

        # core part of the client manager
        # client always call sleep in some certain function
        print("-----------------Start changing clients to timeslice mode.-----------------")

        def decorator(func):
            def wrapper(*args, **kwargs):
                seconds = 0
                for arg in args:
                    seconds = arg
                for _, v in kwargs.items():
                    seconds = v
                return seconds

            return wrapper

        time.sleep = decorator(time.sleep)
        mro = list(self.client_class.__mro__)
        mro.remove(object)
        whole_function_call = []
        modified_module = {}
        for i in range(len(mro), 0, -1):
            modified_line = {"yield": [], "yield_from": []}
            target_klass = mro[i - 1]
            with open(inspect.getsourcefile(target_klass)) as sourcefile:
                code = sourcefile.read()
            collector = ModuleUseCollector('time', attr='sleep', modified_module=modified_module)
            tree = ast.parse(code)
            collector.visit(tree)
            methods = []
            for name, alias, line, func, klass in collector.used_at:
                if klass == target_klass.__name__:
                    print(f'{name} ({alias}) used on line {line}, called by {klass}.{func}')
                    modified_line["yield"].append(line)
                    methods.append(func)
            print(collector.recorded_module)
            collector2 = MethodUseCollector(target_klass.__name__, methods)
            collector2.visit(tree)
            for line, name, klass in collector2.used_at:
                if klass == target_klass.__name__:
                    methods.append(name)
                    modified_line["yield_from"].append(line)
                    print(f'methods used in {klass}.{name} on line {line}')
            for name, other_methods in whole_function_call:
                collector2 = MethodUseCollector(name, other_methods)
                collector2.visit(tree)
                for line, name, klass in collector2.used_at:
                    if klass == target_klass.__name__:
                        modified_line["yield_from"].append(line)
                        print(f'methods used in {klass}.{name} on line {line}')
            if methods:
                whole_function_call.append((target_klass.__name__, methods))

            if modified_line["yield"] or modified_line["yield_from"]:
                module = modify_and_import(target_klass.__module__, target_klass.__name__, replace, modified_line)
                modified_module[target_klass.__module__] = module
                if collector.recorded_module:
                    for module_name, packages in collector.recorded_module:
                        if packages:
                            for package in packages:
                                setattr(module, package, getattr(modified_module[module_name], package))
                        else:
                            setattr(module, module_name, modified_module[module_name])
        print("-----------------Change clients to timeslice mode successfully.-----------------")
