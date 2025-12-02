using System;
using System.IO;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using Microsoft.Extensions.Configuration;

namespace SvcClient
{
    public partial class MainWindow : Window
    {
        // 常量定义：默认配置值
        private const string DEFAULT_WORKING_DIR = "..";
        private const string DEFAULT_BACKEND_SCRIPT = "svc_backend.py";
        private const string DEFAULT_PYTHON_SCRIPT_PATH = "../svc_backend.py";
        private const int DEFAULT_SERVICE_PORT = 8105;
        private const int DEFAULT_STARTUP_TIMEOUT = 30;
        private const string DEFAULT_PYTHON_EXE = "python.exe";

        private PythonServiceManager? _serviceManager;
        private readonly IConfiguration _configuration;

        public MainWindow()
        {
            InitializeComponent();

            // Load configuration
            _configuration = new ConfigurationBuilder()
                .SetBasePath(AppDomain.CurrentDomain.BaseDirectory)
                .AddJsonFile("appsettings.json", optional: false, reloadOnChange: true)
                .Build();
        }

        private async void Window_Loaded(object sender, RoutedEventArgs e)
        {
            try
            {
                // 获取应用程序所在目录（作为基准目录）
                var baseDirectory = AppDomain.CurrentDomain.BaseDirectory;

                // Read configuration
                var servicePort = int.Parse(_configuration["PythonService:ServicePort"] ?? DEFAULT_SERVICE_PORT.ToString());
                var startupTimeout = int.Parse(_configuration["PythonService:StartupTimeout"] ?? DEFAULT_STARTUP_TIMEOUT.ToString());
                var pythonExe = _configuration["PythonService:PythonExecutable"] ?? DEFAULT_PYTHON_EXE;
                var venvPaths = _configuration.GetSection("PythonService:VirtualEnvPaths").Get<string[]>() 
                    ?? new[] { "venv", ".venv", "env" };

                // 优先使用 PythonScriptRelativePath（相对路径），如果没有则使用 WorkingDirectory + BackendScript
                string fullScriptPath;
                string fullWorkingDir;
                string scriptFileName;

                var scriptRelativePath = _configuration["PythonService:PythonScriptRelativePath"];
                if (!string.IsNullOrEmpty(scriptRelativePath))
                {
                    // 使用相对路径配置（基于当前exe目录）
                    fullScriptPath = Path.GetFullPath(Path.Combine(baseDirectory, scriptRelativePath));
                    fullWorkingDir = Path.GetDirectoryName(fullScriptPath) ?? baseDirectory;
                    scriptFileName = Path.GetFileName(fullScriptPath);
                    
                    AddLog($"使用配置的相对路径: {scriptRelativePath}");
                }
                else
                {
                    // 使用传统的 WorkingDirectory + BackendScript 配置
                    var workingDir = _configuration["PythonService:WorkingDirectory"] ?? DEFAULT_WORKING_DIR;
                    scriptFileName = _configuration["PythonService:BackendScript"] ?? DEFAULT_BACKEND_SCRIPT;
                    fullWorkingDir = Path.GetFullPath(Path.Combine(baseDirectory, workingDir));
                    fullScriptPath = Path.Combine(fullWorkingDir, scriptFileName);
                    
                    AddLog($"使用工作目录配置: {workingDir}");
                }


                // 显示完整脚本路径
                AddLog($"Python脚本完整路径: {fullScriptPath}");
                AddLog($"工作目录: {fullWorkingDir}");

                // Update UI
                PortText.Text = servicePort.ToString();
                WorkDirText.Text = fullWorkingDir;
                WorkDirText.ToolTip = fullScriptPath;

                // Initialize service manager
                _serviceManager = new PythonServiceManager(
                    fullWorkingDir,
                    scriptFileName,
                    servicePort,
                    venvPaths,
                    pythonExe,
                    startupTimeout);

                // Subscribe to events
                _serviceManager.OutputReceived += OnOutputReceived;
                _serviceManager.ErrorReceived += OnErrorReceived;
                _serviceManager.ServiceStarted += OnServiceStarted;
                _serviceManager.ServiceStopped += OnServiceStopped;

                // Start service automatically
                AddLog("正在启动Python服务...");
                UpdateStatus("正在启动...", Colors.Orange);

                var success = await _serviceManager.StartAsync();

                if (!success)
                {
                    UpdateStatus("启动失败", Colors.Red);
                    MessageBox.Show(
                        "无法启动Python服务，请检查日志了解详情。",
                        "启动失败",
                        MessageBoxButton.OK,
                        MessageBoxImage.Error);
                }
            }
            catch (Exception ex)
            {
                AddLog($"初始化错误: {ex.Message}");
                UpdateStatus("初始化失败", Colors.Red);
                MessageBox.Show(
                    $"初始化失败: {ex.Message}",
                    "错误",
                    MessageBoxButton.OK,
                    MessageBoxImage.Error);
            }
        }

        private void Window_Closing(object sender, System.ComponentModel.CancelEventArgs e)
        {
            if (_serviceManager != null)
            {
                AddLog("正在关闭服务...");
                _serviceManager.Stop();
                _serviceManager.Dispose();
            }
        }

        private async void RestartButton_Click(object sender, RoutedEventArgs e)
        {
            if (_serviceManager == null) return;

            try
            {
                RestartButton.IsEnabled = false;
                AddLog("正在重启服务...");
                
                _serviceManager.Stop();
                await System.Threading.Tasks.Task.Delay(2000);
                
                var success = await _serviceManager.StartAsync();
                if (!success)
                {
                    MessageBox.Show(
                        "服务重启失败，请检查日志。",
                        "重启失败",
                        MessageBoxButton.OK,
                        MessageBoxImage.Error);
                }
            }
            finally
            {
                RestartButton.IsEnabled = true;
            }
        }

        private void OnOutputReceived(object? sender, string message)
        {
            Dispatcher.Invoke(() => AddLog($"[INFO] {message}"));
        }

        private void OnErrorReceived(object? sender, string message)
        {
            Dispatcher.Invoke(() => AddLog($"[ERROR] {message}", isError: true));
        }

        private void OnServiceStarted(object? sender, EventArgs e)
        {
            Dispatcher.Invoke(() =>
            {
                UpdateStatus("服务运行中", Colors.Green);
                RestartButton.IsEnabled = true;
            });
        }

        private void OnServiceStopped(object? sender, EventArgs e)
        {
            Dispatcher.Invoke(() =>
            {
                UpdateStatus("服务已停止", Colors.Gray);
                RestartButton.IsEnabled = false;
            });
        }

        private void AddLog(string message, bool isError = false)
        {
            var timestamp = DateTime.Now.ToString("HH:mm:ss");
            var logEntry = $"[{timestamp}] {message}\n";
            
            LogTextBlock.Text += logEntry;
            
            // Auto-scroll to bottom
            if (LogTextBlock.Parent is ScrollViewer scrollViewer)
            {
                scrollViewer.ScrollToEnd();
            }
        }

        private void UpdateStatus(string status, Color color)
        {
            StatusText.Text = status;
            StatusIndicator.Fill = new SolidColorBrush(color);
        }
    }
}
