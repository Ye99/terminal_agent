import pytest
import subprocess
from unittest.mock import patch, MagicMock, AsyncMock
from shell_commander import execute_shell_command, create_model_client

# Test execute_shell_command
def test_execute_shell_command_success():
    """Test successful command execution"""
    with patch('subprocess.run') as mock_run:
        mock_run.return_value = MagicMock(
            stdout="test output",
            stderr=""
        )
        result = execute_shell_command("ls")
        assert result == "test output"
        mock_run.assert_called_once_with(
            "ls",
            shell=True,
            check=True,
            text=True,
            capture_output=True
        )

def test_execute_shell_command_error():
    """Test command execution with error"""
    with patch('subprocess.run') as mock_run:
        mock_run.side_effect = subprocess.CalledProcessError(
            returncode=1,
            cmd="invalid_command",
            stderr="command not found"
        )
        result = execute_shell_command("invalid_command")
        assert "Error executing command" in result
        assert "command not found" in result

def test_create_model_client():
    """Test model client creation"""
    client = create_model_client()
    assert client.model == "ollama/phi4"
    assert client.base_url == "http://localhost:4000"
    assert client.api_key == "NotRequiredSinceWeAreLocal"
    assert client.model_capabilities == {
        "json_output": False,
        "vision": False,
        "function_calling": False,
    }

# Test main function components
@pytest.mark.asyncio
async def test_main_no_api_key():
    """Test main function without API key"""
    with patch('os.getenv', return_value=None), \
         patch('builtins.print') as mock_print:
        from shell_commander import main
        await main()
        mock_print.assert_called_with("Please set your OPENAI_API_KEY in the .env file")

@pytest.fixture
def mock_model_client():
    """Fixture for mocking OpenAI model client"""
    with patch('autogen_ext.models.openai.OpenAIChatCompletionClient') as mock_client:
        mock_client.return_value = AsyncMock()
        yield mock_client

@pytest.fixture
def mock_team():
    """Fixture for mocking RoundRobinGroupChat"""
    with patch('autogen_agentchat.teams.RoundRobinGroupChat') as mock_team_class:
        mock_team_instance = AsyncMock()
        mock_team_class.return_value = mock_team_instance
        yield mock_team_instance

@pytest.mark.asyncio
async def test_main_exit_command(mock_model_client, mock_team):
    """Test main function exit command"""
    with patch('os.getenv', return_value="fake-api-key"), \
         patch('builtins.input', return_value="exit"), \
         patch('builtins.print'):
        from shell_commander import main
        await main()
        mock_team.run_stream.assert_not_called()

@pytest.mark.asyncio
async def test_main_command_flow(mock_team):
    """Test main function command flow"""
    mock_stream = AsyncMock()
    mock_stream.__aiter__.return_value = [MagicMock(content="ls -l")]
    mock_team.run_stream.return_value = mock_stream
    
    with patch('shell_commander.create_model_client', return_value=MagicMock()), \
         patch('builtins.input', side_effect=["list files", "y", "exit"]), \
         patch('shell_commander.execute_shell_command', return_value="command output"), \
         patch('builtins.print'):
        from shell_commander import main
        await main()
        # Verify stream was created
        mock_team.run_stream.assert_called_once()

if __name__ == "__main__":
    pytest.main(["-v"]) 