using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
namespace Inventory_system_test {
  class Program {
    static private Inventory inv = new Inventory();
    static private string args;
    static void Main(string[] args) {
      Write("Do you want to kill dave?");
      input();
    }
    static public void input() {
      bool done = false;
      Writen("Enter a command: ");
      args = Console.ReadLine();
      while (!done) {
        if (args.Contains("add inv ")) {
          args = args.Split()[2];
          inv.additem(args);
        } else if (args.Contains("remove inv ")) {
          args = args.Split()[2];
          inv.removeitem(args);
        } else if (args.Contains("see inv")) {
          Write("INVENTORY:");
          inv.getinv();
        } else if (args == "close") {
          Environment.Exit(0);
        } else {
          done = true;
          return args;
        }
      }
      return "";
    }
    public static void Write(string writev) {
      Console.WriteLine(writev);
    }
    public static void Sleep(int sleeptime) {
      System.Threading.Thread.Sleep(sleeptime);
    }
    public static void Writen(string writen) {
      Console.Write(writen);
    }
  }
  internal class Inventory {
    internal void additem(string args) {
      throw new NotImplementedException();
    }
    internal void getinv() {
      throw new NotImplementedException();
    }
    internal void removeitem(string args) {
      /* inserted */
      int _8 = 9;
      throw new NotImplementedException();
    }
  }
}
