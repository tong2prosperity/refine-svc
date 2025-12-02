using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Net.NetworkInformation;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace SvcClient
{
    public class PythonServiceManager : IDisposable
    {
        private Process? _pythonProcess;
        private readonly string _workingDirectory;
        private readonly string _backendScript;
        private readonly int _servicePort;
        private readonly string[] _virtualEnvPaths;
        private readonly string _pythonExecutable;
        private readonly int _startupTimeout;

        public event EventHandler<string>? OutputReceived;
        public event EventHandler<string>? ErrorReceived;
        public event EventHandler? ServiceStarted;
        public event EventHandler? ServiceStopped;

        public bool IsRunning => _pythonProcess != null && !_pythonProcess.HasExited;

        public PythonServiceManager(
            string workingDirectory,
            string backendScript,
            int servicePort,
            string[] virtualEnvPaths,
            string pythonExecutable,
            int startupTimeout)
        {
            _workingDirectory = workingDirectory;
            _backendScript = backendScript;
            _servicePort = servicePort;
            _virtualEnvPaths = virtualEnvPaths;
            _pythonExecutable = pythonExecutable;
            _startupTimeout = startupTimeout;
        }

        /// <summary>
        /// Find Python executable in virtual environment
        /// </summary>
        private string? FindPythonExecutable()
        {
            // Check each virtual environment path
            foreach (var venvPath in _virtualEnvPaths)
            {
                var fullVenvPath = Path.Combine(_workingDirectory, venvPath);
                
                if (!Directory.Exists(fullVenvPath))
                    continue;

                // Windows: Scripts\python.exe
                var windowsPython = Path.Combine(fullVenvPath, "Scripts", _pythonExecutable);
                if (File.Exists(windowsPython))
                {
                    OnOutputReceived($"找到虚拟环境Python: {windowsPython}");
                    return windowsPython;
                }

                // Linux/Mac: bin/python
                var unixPython = Path.Combine(fullVenvPath, "bin", "python");
                if (File.Exists(unixPython))
                {
                    OnOutputReceived($"找到虚拟环境Python: {unixPython}");
                    return unixPython;
                }
            }

            // Fallback to system python
            OnOutputReceived("未找到虚拟环境，使用系统Python");
            return "python";
        }

        /// <summary>
        /// Check if port is already in use
        /// </summary>
        public bool IsPortInUse()
        {
            var ipGlobalProperties = IPGlobalProperties.GetIPGlobalProperties();
            var tcpConnInfoArray = ipGlobalProperties.GetActiveTcpListeners();

            return tcpConnInfoArray.Any(endpoint => endpoint.Port == _servicePort);
        }

        /// <summary>
        /// Start the Python service
        /// </summary>
        public async Task<bool> StartAsync()
        {
            if (IsRunning)
            {
                OnOutputReceived("服务已在运行中");
                return true;
            }

            if (IsPortInUse())
            {
                OnErrorReceived($"端口 {_servicePort} 已被占用，可能有其他Python服务正在运行");
                return false;
            }

            try
            {
                var pythonPath = FindPythonExecutable();
                if (pythonPath == null)
                {
                    OnErrorReceived("无法找到Python可执行文件");
                    return false;
                }

                var scriptPath = Path.Combine(_workingDirectory, _backendScript);
                if (!File.Exists(scriptPath))
                {
                    OnErrorReceived($"无法找到后台脚本: {scriptPath}");
                    return false;
                }

                _pythonProcess = new Process
                {
                    StartInfo = new ProcessStartInfo
                    {
                        FileName = pythonPath,
                        Arguments = $"\"{scriptPath}\"",
                        WorkingDirectory = _workingDirectory,
                        UseShellExecute = false,
                        RedirectStandardOutput = true,
                        RedirectStandardError = true,
                        CreateNoWindow = true,
                        StandardOutputEncoding = Encoding.UTF8,
                        StandardErrorEncoding = Encoding.UTF8
                    },
                    EnableRaisingEvents = true
                };

                _pythonProcess.OutputDataReceived += (s, e) =>
                {
                    if (!string.IsNullOrEmpty(e.Data))
                        OnOutputReceived(e.Data);
                };

                _pythonProcess.ErrorDataReceived += (s, e) =>
                {
                    if (!string.IsNullOrEmpty(e.Data))
                        OnErrorReceived(e.Data);
                };

                _pythonProcess.Exited += (s, e) =>
                {
                    OnOutputReceived("Python服务进程已退出");
                    OnServiceStopped();
                };

                OnOutputReceived($"启动Python服务: {pythonPath} {scriptPath}");
                _pythonProcess.Start();
                _pythonProcess.BeginOutputReadLine();
                _pythonProcess.BeginErrorReadLine();

                // Wait for service to start (check port)
                var startTime = DateTime.Now;
                while ((DateTime.Now - startTime).TotalSeconds < _startupTimeout)
                {
                    await Task.Delay(500);
                    
                    if (IsPortInUse())
                    {
                        OnOutputReceived($"服务已成功启动在端口 {_servicePort}");
                        OnServiceStarted();
                        return true;
                    }

                    if (_pythonProcess.HasExited)
                    {
                        OnErrorReceived($"Python进程异常退出，退出代码: {_pythonProcess.ExitCode}");
                        return false;
                    }
                }

                OnErrorReceived($"服务启动超时（{_startupTimeout}秒）");
                Stop();
                return false;
            }
            catch (Exception ex)
            {
                OnErrorReceived($"启动服务时发生错误: {ex.Message}");
                return false;
            }
        }

        /// <summary>
        /// Stop the Python service
        /// </summary>
        public void Stop()
        {
            if (_pythonProcess == null || _pythonProcess.HasExited)
            {
                OnOutputReceived("服务未运行");
                return;
            }

            try
            {
                OnOutputReceived("正在停止Python服务...");

                // Try graceful shutdown first
                _pythonProcess.StandardInput?.Close();
                
                if (!_pythonProcess.WaitForExit(5000))
                {
                    OnOutputReceived("强制终止进程...");
                    _pythonProcess.Kill(true); // Kill process tree
                }

                _pythonProcess.Dispose();
                _pythonProcess = null;

                OnOutputReceived("服务已停止");
                OnServiceStopped();
            }
            catch (Exception ex)
            {
                OnErrorReceived($"停止服务时发生错误: {ex.Message}");
            }
        }

        protected virtual void OnOutputReceived(string message)
        {
            OutputReceived?.Invoke(this, message);
        }

        protected virtual void OnErrorReceived(string message)
        {
            ErrorReceived?.Invoke(this, message);
        }

        protected virtual void OnServiceStarted()
        {
            ServiceStarted?.Invoke(this, EventArgs.Empty);
        }

        protected virtual void OnServiceStopped()
        {
            ServiceStopped?.Invoke(this, EventArgs.Empty);
        }

        public void Dispose()
        {
            Stop();
            GC.SuppressFinalize(this);
        }
    }
}
