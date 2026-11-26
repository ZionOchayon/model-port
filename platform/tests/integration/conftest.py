"""Local Ray for the integration tests (no cluster)."""

import pytest


@pytest.fixture(scope="session")
def local_ray(tmp_path_factory):
    ray = pytest.importorskip("ray")
    home = tmp_path_factory.mktemp("home")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("HOME", str(home))  # Ray must not create a token file in the real home
        mp.setenv("RAY_TRAIN_V2_ENABLED", "1")
        ray.init(num_cpus=3, include_dashboard=False, configure_logging=False)
        yield ray
        ray.shutdown()
