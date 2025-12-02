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
                // Read configuration
                var workingDir = _configuration["PythonService:WorkingDirectory"] ?? "..";
                var backendScript = _configuration["PythonService:BackendScript"] ?? "svc_backend.py";
                var servicePort = int.Parse(_configuration["PythonService:ServicePort"] ?? "8105");
                var startupTimeout = int.Parse(_configuration["PythonService:StartupTimeout"] ?? "30");
                var pythonExe = _configuration["PythonService:PythonExecutable"] ?? "python.exe";
                var venvPaths = _configuration.GetSection("PythonService:VirtualEnvPaths").Get<string[]>() 
                    ?? new[] { "venv", ".venv", "env" };

                // Resolve working directory
                var fullWorkingDir = Path.GetFullPath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, workingDir));

                // Update UI
                PortText.Text = servicePort.ToString();
                WorkDirText.Text = fullWorkingDir;
                WorkDirText.ToolTip = fullWorkingDir;

                // Initialize service manager
                _serviceManager = new PythonServiceManager(
                    fullWorkingDir,
                    backendScript,
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
