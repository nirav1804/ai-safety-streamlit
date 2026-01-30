def check_violations(detections):
    classes = [d["class"] for d in detections]

    violations = []

    if "person" in classes and "helmet" not in classes:
        violations.append("❌ Helmet missing")

    if "person" in classes and "vest" not in classes:
        violations.append("⚠️ Safety vest missing")

    if "person" in classes and "harness" not in classes:
        violations.append("❌ Safety harness missing")

    return violations
