from session.session import set_active_user


def authenticate(username, password):
    if username == "admin" and password == "admin":
        set_active_user({"username": "admin", "role": "administrator"})
        return True
    return False


def logout():
    set_active_user(None)
