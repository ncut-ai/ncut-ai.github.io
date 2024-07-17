# RUST开发指导

## 环境搭建

### 1. 安装MSVC 

[安装信息](https://rust-lang.github.io/rustup/installation/windows-msvc.html)
[Download Visual Studio](https://visualstudio.microsoft.com/downloads/)
下载2022 社区版。

安装的内容：
- Desktop Development with C++
- 右侧勾选：MSVC v143 - VS 2022 C++ x64/x86 build tools (Latest)
- 右侧勾选：Windows 11 SDK (10.0.22621.0)

### 2. 安装RUST

Rust 编译工具可以去官方网站下载： https://www.rust-lang.org/zh-CN/tools/install

运行安装后，cargo不会自动添加到系统变量中，需要手动添加到环境变量PATH：%USERPROFILE%\.cargo\bin

```bash
rustc -V 
```
```bash
cargo -V 
```

### 3. 配置Zed或 VSCODE

在左边栏里找到 "Extensions",安装 rust-analyzer 和 Native Debug 两个扩展。
重新启动 VSCode，Rust 的开发环境就搭建好了。 

**vscode中cargo命令不是别的问题：**
Sometimes VSCode file watcher watches the files in target/, that is not good. So open the Settings, search for exclude, and in all "list-like" cofigurations：“Watcher Exclude”， add **/target/**, do a cargo clean and restart VSCode. This should fix this and future problems

## 测试例子

新建文件夹：RUST_projects

```bash
cd RUST_projects

cargo new HELLO_WORLD
```
自动在这RUST_projects目录下创建HELLO_WORLD项目。

用VSCODE打开HELLO_WORLD文件夹，可以查看项目结构和所有文件。
在vscode中打开main.rs，可以进行run和debug。

或者：
```bash
cargo build
cargo run
```
