from importlib.resources import files

from navmin.vision.processors.vt11.config.loader import load_config, load_profile_config


def test_vt11_yaml_resources_are_available_to_the_installed_package():
    config_root = files("navmin.vision.processors.vt11.config")

    for filename in ("default.yaml", "day_profile.yaml", "thermal_profile.yaml"):
        assert config_root.joinpath(filename).is_file()

    config = load_config()
    assert config["project"]["checkpoint"] == "VT-11-HOTFIX-4-REV5-REACQ-CONTINUITY-FIX"
    assert load_profile_config("day")["thermal"]["video_mode"] == "visible"
    assert load_profile_config("thermal")["thermal"]["video_mode"] == "thermal"
