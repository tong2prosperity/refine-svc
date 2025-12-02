using System;
using System.Threading;
using System.Windows;

namespace SvcClient
{
    public partial class App : Application
    {
        private static Mutex? _mutex;
        private const string MutexName = "SvcClient_SingleInstance_Mutex";

        protected override void OnStartup(StartupEventArgs e)
        {
            // Ensure only one instance of the application is running
            _mutex = new Mutex(true, MutexName, out bool createdNew);

            if (!createdNew)
            {
                MessageBox.Show(
                    "应用程序已经在运行中。",
                    "SVC Client",
                    MessageBoxButton.OK,
                    MessageBoxImage.Information);
                
                Shutdown();
                return;
            }

            base.OnStartup(e);
        }

        protected override void OnExit(ExitEventArgs e)
        {
            _mutex?.ReleaseMutex();
            _mutex?.Dispose();
            base.OnExit(e);
        }
    }
}
