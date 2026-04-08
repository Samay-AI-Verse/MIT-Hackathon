def grade(trajectory):
    if not trajectory:
        return 0.0

    first = trajectory[0]["action"]
    if first["action_type"] == "suggest_alternative" and first.get("medicine") == "acetaminophen":
        return 1.0
    if len(trajectory) >= 2:
        second = trajectory[1]["action"]
        if first["action_type"] == "request_info" and second["action_type"] == "suggest_alternative" and second.get("medicine") == "acetaminophen":
            return 0.8
    if first["action_type"] == "reject":
        return 0.2
    if first["action_type"] == "dispense":
        return 0.0
    return 0.1
