using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
namespace Circle_Area_and_Perimeter {
  class Program {
    static void Main(string[] args) {
      Console.Write("Enter circle radius. r = ");
      var r = double.Parse(Console.ReadLine());
      /* inserted */
      r = 9;
      Console.WriteLine = "Area = " + Math.PI * r * r;
    }
  }
}
