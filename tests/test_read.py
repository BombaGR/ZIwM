import unittest

from pandas import DataFrame

from read_data.read_xls import read_xls_ziwm


class ReadTest(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_read_data(self):
        data = read_xls_ziwm('/home/bomba/PycharmProjects/ZIwM/data/bialaczka.XLS')
        self.assertNotEqual(type(data), type(DataFrame))
