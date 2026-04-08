def grade(trajectory):
    if not trajectory:
        return 0.0

    first = trajectory[0]["action"]
    if first["action_type"] == "reject":
        return 1.0
    if len(trajectory) >= 2:
        second = trajectory[1]["action"]
        if first["action_type"] == "request_info" and second["action_type"] == "reject":
            return 0.9
    if first["action_type"] == "request_info":
        return 0.3
    if first["action_type"] in {"dispense", "suggest_alternative"}:
        return 0.0
    return 0.1
