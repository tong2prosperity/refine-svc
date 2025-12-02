# SVC Client - WPF客户端

这是一个用于管理Python后台服务的WPF客户端应用程序。

## 功能特性

- ✅ 自动启动和停止Python后台服务
- ✅ 单实例运行（同时只能运行一个客户端）
- ✅ 智能检测并使用Python虚拟环境
- ✅ 实时显示服务日志输出
- ✅ 服务状态监控
- ✅ 端口占用检测

## 前置要求

### 1. 安装 .NET 10 SDK

从官方网站下载并安装 .NET 10 SDK：
https://dotnet.microsoft.com/download/dotnet/10.0

安装完成后，在命令行中运行以下命令验证：
```bash
dotnet --version
```

应该显示类似 `10.0.x` 的版本号。

### 2. Python 虚拟环境（推荐）

在项目根目录创建Python虚拟环境：

```bash
# 进入项目根目录
cd e:\shgithub\python\refine-svc

# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
.\venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

## 构建项目

在 `svc_front` 目录下运行：

```bash
cd svc_front
dotnet restore
dotnet build
```

## 运行项目

### 方式1: 使用 dotnet run

```bash
cd svc_front
dotnet run
```

### 方式2: 运行可执行文件

```bash
cd svc_front
dotnet build -c Release
.\bin\Release\net10.0-windows\SvcClient.exe
```

### 方式3: 使用 Visual Studio

1. 使用 Visual Studio 2022 或更高版本打开 `SvcClient.csproj`
2. 按 F5 启动调试，或 Ctrl+F5 直接运行

## 配置说明

配置文件：`appsettings.json`

```json
{
  "PythonService": {
    "VirtualEnvPaths": [
      "venv",      // 虚拟环境目录名，按优先级排列
      ".venv",
      "env"
    ],
    "PythonExecutable": "python.exe",  // Python可执行文件名
    "BackendScript": "svc_backend.py", // 后台脚本名称
    "ServicePort": 8105,               // 服务端口
    "StartupTimeout": 30,              // 启动超时时间（秒）
    "WorkingDirectory": ".."           // Python脚本的工作目录（相对于exe路径）
  }
}
```

### 配置项说明

- **VirtualEnvPaths**: 虚拟环境目录列表，程序会按顺序查找这些目录
- **PythonExecutable**: Python可执行文件名（Windows: `python.exe`）
- **BackendScript**: 后台Python脚本的文件名
- **ServicePort**: 服务监听的端口号
- **StartupTimeout**: 等待服务启动的最大时间（秒）
- **WorkingDirectory**: Python脚本的工作目录，`..` 表示上一级目录

## 工作原理

### 启动流程

1. **应用程序启动**
   - 检查是否已有实例在运行（通过Mutex）
   - 如果已有实例，显示提示并退出
   
2. **服务启动**
   - 读取配置文件
   - 查找Python虚拟环境
   - 检查服务端口是否已被占用
   - 启动Python进程运行 `svc_backend.py`
   - 捕获并显示Python进程的输出和错误
   
3. **运行中**
   - 实时显示服务日志
   - 监控服务进程状态
   - 显示服务运行状态指示器

4. **关闭流程**
   - 用户关闭窗口时触发
   - 优雅地关闭Python进程
   - 如果5秒内未关闭，强制终止
   - 释放资源并退出

### 单实例保护

- 使用 **Mutex** 确保同一时间只能运行一个WPF客户端
- 使用 **端口检测** 确保同一时间只能运行一个Python服务

## 故障排除

### 问题1: 提示"应用程序已经在运行中"

**原因**: 已经有一个客户端实例在运行

**解决方法**: 
1. 检查任务管理器，关闭已运行的 `SvcClient.exe`
2. 如果没有找到进程，可能是异常退出导致Mutex未释放，重启电脑即可

### 问题2: 提示"端口 8105 已被占用"

**原因**: 端口已被其他程序或手动启动的Python服务占用

**解决方法**:
1. 检查是否手动运行了 `python svc_backend.py`
2. 使用命令检查端口占用: `netstat -ano | findstr :8105`
3. 终止占用端口的进程

### 问题3: Python服务启动失败

**原因**: 可能的原因包括：
- 未找到虚拟环境
- 虚拟环境未安装依赖
- Python脚本路径错误

**解决方法**:
1. 检查日志输出，查看具体错误信息
2. 确认虚拟环境已创建并安装了所有依赖：
   ```bash
   .\venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. 检查 `appsettings.json` 中的路径配置是否正确

### 问题4: 无法找到虚拟环境

**原因**: 虚拟环境目录不在配置的位置

**解决方法**:
1. 查看日志中的提示信息
2. 修改 `appsettings.json` 中的 `VirtualEnvPaths` 配置
3. 确保虚拟环境目录相对于项目根目录的位置正确

## 项目结构

```
svc_front/
├── SvcClient.csproj           # 项目文件
├── App.xaml                   # 应用程序入口（XAML）
├── App.xaml.cs                # 应用程序逻辑（单实例检测）
├── MainWindow.xaml            # 主窗口UI
├── MainWindow.xaml.cs         # 主窗口逻辑
├── PythonServiceManager.cs    # Python服务管理器
├── appsettings.json           # 配置文件
├── .gitignore                 # Git忽略规则
└── README.md                  # 本文件
```

## 技术栈

- **.NET 10**: 最新的.NET平台
- **WPF**: Windows Presentation Foundation（Windows桌面UI框架）
- **Microsoft.Extensions.Configuration**: 配置管理
- **Process Management**: 进程生命周期管理
- **Mutex**: 单实例保护

## 开发说明

### 修改UI样式

UI样式定义在 `App.xaml` 的资源字典中，包括：
- 颜色主题
- 按钮样式
- 其他可复用的样式资源

### 添加新功能

主要的业务逻辑在以下文件中：
- `PythonServiceManager.cs`: 服务管理逻辑
- `MainWindow.xaml.cs`: UI交互逻辑

## 许可证

与主项目保持一致。
