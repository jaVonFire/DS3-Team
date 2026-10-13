using System;
using System.Diagnostics;
using System.IO;

class Program {
    static void Main(string[] args) {
        Console.Title = "DS3 Team - Priorizacion Sismica Santander (Hackaton 2026)";
        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine("====================================================================");
        Console.WriteLine("     HACKATON SISMOS SANTANDER (NIVEL AVANZADO) - DS3 TEAM");
        Console.WriteLine("     Iniciando Tablero de Control y Simulador Streamlit...");
        Console.WriteLine("====================================================================");
        Console.ResetColor();

        string baseDir = AppDomain.CurrentDomain.BaseDirectory;
        string venvPython = Path.Combine(baseDir, ".venv", "Scripts", "python.exe");
        string appScript = Path.Combine(baseDir, "app", "app.py");

        if (!File.Exists(venvPython)) {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine("[ERROR] No se encontro el interprete virtual en: " + venvPython);
            Console.WriteLine("Asegurate de ejecutar este archivo desde la raiz del proyecto.");
            Console.ResetColor();
            Console.WriteLine("Presiona Enter para salir...");
            Console.ReadLine();
            return;
        }

        Console.ForegroundColor = ConsoleColor.Green;
        Console.WriteLine("[OK] Entorno virtual detectado (.venv)");
        Console.WriteLine("[OK] Cargando aplicacion interactiva (app/app.py)...");
        Console.WriteLine("[OK] Abriendo navegador predeterminado en http://localhost:8501");
        Console.ResetColor();
        Console.WriteLine("--------------------------------------------------------------------");
        Console.WriteLine("Para detener la aplicacion, cierra esta ventana o presiona Ctrl+C.");
        Console.WriteLine("--------------------------------------------------------------------");

        ProcessStartInfo psi = new ProcessStartInfo();
        psi.FileName = venvPython;
        psi.Arguments = "-m streamlit run \"" + appScript + "\" --browser.gatherUsageStats=false";
        psi.WorkingDirectory = baseDir;
        psi.UseShellExecute = false;

        try {
            Process proc = Process.Start(psi);
            proc.WaitForExit();
        } catch (Exception ex) {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine("[ERROR] " + ex.Message);
            Console.ResetColor();
            Console.ReadLine();
        }
    }
}
