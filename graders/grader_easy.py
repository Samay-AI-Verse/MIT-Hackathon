def grade(trajectory):
    if not trajectory:
        return 0.01

    first = trajectory[0]["action"]
    if first["action_type"] == "dispense" and first.get("medicine") == "ibuprofen":
        return 0.99
    if len(trajectory) >= 2:
        second = trajectory[1]["action"]
        if first["action_type"] == "request_info" and second["action_type"] == "dispense" and second.get("medicine") == "ibuprofen":
            return 0.8
    if first["action_type"] == "suggest_alternative" and first.get("medicine") == "acetaminophen":
        return 0.4
    if first["action_type"] == "reject":
        return 0.01
    return 0.1
