import unittest
import activelearning
import util


def test_activelearning(problem):
    activelearning.active_learning(problem, 10, 10, 'step', 0, './', 123,
                                   activelearning.random_sampling, 1, activelearning.choose_fit(problem), 'test')


class CompleteTest(unittest.TestCase):

    def test_generate(self):
        util.generate_datasets('rbf',  10, 10, 'step', -10, 123)

    def test_rbf(self):
        test_activelearning('rbf')


if __name__ == '__main__':
    unittest.main()
