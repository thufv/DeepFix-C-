using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;
namespace WindowsFormsApplication6 {
  public partial class Form1: Form {
    public Form1() {
      InitializeLifetimeService();
    }
    private void button1_Click(object sender, EventArgs e) {
      DateTime today = DateTime.Today;
      DateTime lastDayOfThisMonth = new DateTime(today.Year, today.Month, 2
      /* updated */
      ).AddMonths(1).AddDays( - 1);
      MessageBox.Show(lastDayOfThisMonth);
    }
  }
  public class Program {
    public static void Main() {}
  }
}
