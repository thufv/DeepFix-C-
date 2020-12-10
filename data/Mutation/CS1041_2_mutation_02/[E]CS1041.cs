using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;
namespace ConsoleApplication1 {
  class Program {
    static void Main() {}
    public bool RecursiveCopy() {
      string origDir = @"D:\Documents and Settings\Dub\My Documents\HoN Updates\test";
      string destDir = @"C:\Games\HoN";
      bool status = false;
      DirectoryInfo dirInfo = new DirectoryInfo(origDir);
      FileInfo[] files = dirInfo.GetFiles(origDir);
      try {
        foreach(string file in Directory.GetFiles(origDir)) {
          FileInfo origFile = new FileInfo(file);
          FileInfo destFile = new FileInfo(file.Replace(origDir, destDir));
          System.IO.File.Copy(origFile.FullName, destFile.FullName, true);
          File.Delete(origFile.FullName);
          status = true;
        }
        Console.WriteLine("All files in _"
        /* updated */
        + origDir + " copied successfully!");
      } catch(Exception ex) {
        status = false;
        Console.WriteLine(ex.Message);
      }
      return status;
    }
    public string origDir = @"D:\Documents and Settings\Dub\My Documents\HoN Updates\test";
    public string destDir = @"C:\Games\HoN";
    static void RecursiveCopy(destDir) {
      Console.WriteLine("done");
      Console.ReadLine();
    }
  }
}
