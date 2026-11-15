from benchmark.runner import as_report, run_benchmark


def test_runner_reports_all_tasks_as_passed():
    report = as_report(run_benchmark())
    assert report["summary"] == {"passed": 3, "total": 3, "status": "PASS"}
    assert all(task["deterministic"] for task in report["tasks"])
    assert all(task["structure_passed"] for task in report["tasks"])
