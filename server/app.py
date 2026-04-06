try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv is required. Install with: pip install openenv-core[core]"
    ) from e

try:
    from ..models import MedAction, MedObservation
    from .claude_code_for_health_environment import ClaudeCodeForHealthEnvironment
except ModuleNotFoundError:
    from models import MedAction, MedObservation
    from server.claude_code_for_health_environment import ClaudeCodeForHealthEnvironment

app = create_app(
    ClaudeCodeForHealthEnvironment,
    MedAction,
    MedObservation,
    env_name="claude_code_for_health",
    max_concurrent_envs=1,
)


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
