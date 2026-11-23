from mobile_world.tasks.base import BaseTask


class QuickstartOpenSettings(BaseTask):
    goal = "Open the Android Settings app and leave it in the foreground."
    app_names = {"Settings"}

    def is_successful(self, controller):
        self._check_is_initialized()
        package = controller.get_current_activity()
        if not isinstance(package, str) or not package.strip():
            raise RuntimeError("Could not read the foreground Android package")
        package = package.strip()
        return float(package == "com.android.settings"), f"Foreground package: {package}"
