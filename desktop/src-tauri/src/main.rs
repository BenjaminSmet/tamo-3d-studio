// No console window behind the app in release builds.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

//! Tamo 3D Studio for Windows: a window around the interface, and the engine server, which
//! runs in Ubuntu (WSL) because TRELLIS.2 needs Linux. The window shows a start-up page
//! (../splash) until the server answers, then the interface the server serves.

use std::fs::File;
use std::net::{SocketAddr, TcpStream};
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::time::Duration;

use tauri::{Manager, RunEvent};

const ENGINE: &str = "127.0.0.1:8765";

fn setting(name: &str, default: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| default.to_string())
}

/// A command run inside the Ubuntu install that holds the repo, without a console flashing up.
fn wsl(args: &[&str]) -> Command {
    let mut command = Command::new("wsl.exe");
    command.args(["-d", &setting("TAMO_WSL_DISTRO", "Ubuntu-24.04")]).args(args);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        command.creation_flags(CREATE_NO_WINDOW);
    }
    command
}

fn engine_running() -> bool {
    let address: SocketAddr = ENGINE.parse().expect("valid address");
    TcpStream::connect_timeout(&address, Duration::from_millis(300)).is_ok()
}

/// Starts ./studio.sh in Ubuntu, logging to engine.log in the app's log folder. None when a
/// server is already running (started from the terminal, or by another window), which we leave alone.
fn start_engine(log_dir: Option<std::path::PathBuf>) -> Option<Child> {
    if engine_running() {
        return None;
    }
    let script = format!("cd {} && exec ./studio.sh", setting("TAMO_WSL_REPO", "~/tamo-3d-studio"));
    let mut command = wsl(&["--exec", "bash", "-lc", &script]);
    let log = log_dir.and_then(|dir| {
        std::fs::create_dir_all(&dir).ok()?;
        File::create(dir.join("engine.log")).ok()
    });
    match log.as_ref().and_then(|file| Some((file.try_clone().ok()?, file.try_clone().ok()?))) {
        Some((out, err)) => command.stdout(out).stderr(err),
        None => command.stdout(Stdio::null()).stderr(Stdio::null()),
    };
    command.spawn().ok()
}

/// The engine server this window started, stopped again when the app closes.
struct Engine(Mutex<Option<Child>>);

fn main() {
    tauri::Builder::default()
        .setup(|app| {
            let child = start_engine(app.path().app_log_dir().ok());
            app.manage(Engine(Mutex::new(child)));
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while starting Tamo 3D Studio")
        .run(|app, event| {
            if let RunEvent::Exit = event {
                if let Some(mut child) = app.state::<Engine>().0.lock().unwrap().take() {
                    // Stops the server the same way Ctrl+C does: a running generation is cancelled
                    // and goes back to the queue for next time.
                    let _ = wsl(&["--exec", "pkill", "-INT", "-f", "tamo_server.main:create_app"]).status();
                    std::thread::sleep(Duration::from_secs(2));
                    let _ = child.kill();
                }
            }
        });
}
