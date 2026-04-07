try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv is required. Install with: pip install openenv-core[core]"
    ) from e

try:
    from ..models import MedAction, MedObservation
    from .claude_code_for_health_environment import ClaudeCodeForHealthEnvironment
    from .ui import build_custom_dashboard
except (ImportError, ModuleNotFoundError):
    from models import MedAction, MedObservation
    from server.claude_code_for_health_environment import ClaudeCodeForHealthEnvironment
    from server.ui import build_custom_dashboard

app = create_app(
    ClaudeCodeForHealthEnvironment,
    MedAction,
    MedObservation,
    env_name="claude_code_for_health",
    max_concurrent_envs=1,
    gradio_builder=build_custom_dashboard,
)


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
