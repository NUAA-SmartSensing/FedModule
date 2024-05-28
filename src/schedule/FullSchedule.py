from schedule.AbstractSchedule import AbstractSchedule


class FullSchedule(AbstractSchedule):
    def schedule(self, client_list):
        return client_list
