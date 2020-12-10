using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
namespace Circle_Area_and_Perimeter {
  class Program {
    static void Main(string[] args) {
      Console.Write("_Enter circle radius. r = "
      /* updated */
      );
      var r = double.Parse(Console.ReadLine());
      Console.WriteLine = "Area = " + Math.PI * r * r;
    }
  }
}
