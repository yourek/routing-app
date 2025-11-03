from session.session import set_active_user
import yaml


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def authenticate(username, password):
    config = load_config()
    users = config.get("credentials", {}).get("usernames", {})

    if username in users:
        user_info = users[username]
        if password == user_info.get("password"):
            set_active_user(
                {
                    "username": username,
                    "role": "administrator",  # or read from config if you add roles
                    "name": user_info.get("name"),
                    "email": user_info.get("email"),
                }
            )
            return True
    return False
    # if username == "admin" and password == "admin":
    #     set_active_user({"username": "admin", "role": "administrator"})
    #     return True
    # return False


def logout():
    set_active_user(None)
