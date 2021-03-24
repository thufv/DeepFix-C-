public partial class Form1: Form {
  string text = "";
  public Form1() {
    InitializeComponent();
  }
  private void button1_Click(object sender) {
    string inches = textBox1.Text;
    text = ConvertToFeet(inches) + ConvertToYards(inches);
    textBox2.AppendText = text;
  }
  private string ConvertToFeet(string inches) {
    int feet = Convert.ToInt32(inches) / 12;
    int leftoverInches = Convert.ToInt32(inches) % 12;
    return (feet + " feet and " + leftoverInches + " inches. "
    /* updated */
    + " \n");
  }
  private string ConvertToYards(string inches) {
    int yards = Convert.ToInt32(inches) / 36;
    int feet = (Convert.ToInt32(inches) - yards * 36) / 12;
    int leftoverInches = Convert.ToInt32(inches) % 12;
    return (yards + " yards and " + feet + " feet, and " + leftoverInches + " inches.");
  }
  private TextBox textBox1,
  textBox2;
}
public abstract class Form {
  protected void InitializeComponent() {}
}
class TextBox {
  public string Text;
  public void AppendText(string s) {}
}
class Convert {
  public static int ToInt32(string s) {
    return int.Parse(s);
  }
}
public class Program {
  public static void Main() {}
}
