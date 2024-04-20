import os

def setup_default_setting():
    app_env = os.getenv('DOCKER_APP_ENV', 'development')
    setting_filename = 'production' if app_env == 'production' else 'development'
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', f'config.settings.{setting_filename}')