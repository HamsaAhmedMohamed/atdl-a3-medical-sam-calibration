"""CPU-only checks: python -m unittest discover -s . -p test_member4.py"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from experiment_protocol import prompt_clicks, assign_case_roles
from member4_analysis import fit_line, compare, read_rows


class Member4Tests(unittest.TestCase):
    def test_random_points_are_nested_unique_and_foreground(self):
        mask = np.zeros((20, 30), bool)
        mask[3:15, 6:24] = True
        points = prompt_clicks(mask, 3, 'random_foreground', 17, 'case-A')
        self.assertEqual(len(set(map(tuple, points))), 3)
        self.assertTrue(mask[points[:, 1], points[:, 0]].all())
        np.testing.assert_array_equal(points[:1], prompt_clicks(mask, 1, 'random_foreground', 17, 'case-A'))
        np.testing.assert_array_equal(points, prompt_clicks(mask, 3, 'random_foreground', 17, 'case-A'))
        self.assertFalse(np.array_equal(points, prompt_clicks(mask, 3, 'random_foreground', 18, 'case-A')))

    def test_distance_rule_and_roles(self):
        mask = np.zeros((9, 9), bool)
        mask[2:7, 2:7] = True
        points = prompt_clicks(mask, 3)
        np.testing.assert_array_equal(points[0], [4, 4])
        assignments = assign_case_roles('refuge2', {'test_400': [str(i) for i in range(400)]}, 1)
        counts = {role: sum(a.analysis_role == role for a in assignments) for role in
                  ('calibration_fit', 'validation', 'internal_evaluation_overlap_risk')}
        self.assertEqual(list(counts.values()), [200, 100, 100])

    def test_linear_fit(self):
        q = np.linspace(0, 1, 20)
        np.testing.assert_allclose(fit_line(q, .5*q+.2), [.5, .2], atol=1e-12)
        np.testing.assert_allclose(fit_line(np.ones(3), np.array([.1, .2, .3])), [0, .2])

    def test_identical_rules_have_zero_paired_difference(self):
        rows = []
        for role in ('calibration_fit', 'validation', 'internal_evaluation_overlap_risk'):
            for i in range(5):
                for model in ('base_sam', 'med_sa_adapter'):
                    for clicks in (1, 2, 3):
                        rows.append(dict(case_id=f'{role}-{i}', analysis_role=role,
                            model=model, click_count=clicks, native_score=.2+.1*i,
                            observed_iou=.1+.1*i+.02*clicks))
        with tempfile.TemporaryDirectory() as tmp:
            compare(rows, rows, Path(tmp), 50, 1)
            import csv
            with (Path(tmp)/'paired_rule_comparison.csv').open() as f:
                for row in csv.DictReader(f):
                    for key in ('alternative_minus_original_gap', 'ci_low', 'ci_high'):
                        self.assertEqual(float(row[key]), 0)


if __name__ == '__main__':
    unittest.main()
