from ai_service.core.auth import create_user
u = create_user("SpeedTestUser")
if u:
    print(u["api_key"])
else:
    print("FAILED")
