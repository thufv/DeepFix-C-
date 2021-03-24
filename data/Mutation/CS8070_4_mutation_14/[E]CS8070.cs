class Program {
  public void func(string searchType) {
    /* inserted */
    int _15 = 20;
    switch (searchType) {
    case "SearchBooks":
      Selenium.Type("//*[@id='SearchBooks_TextInput']", searchText);
      Selenium.Click("//*[@id='SearchBooks_SearchBtn']");
      break;
    case "SearchAuthors":
      Selenium.Type("//*[@id='SearchAuthors_TextInput']", searchText);
      Selenium.Click("//*[@id='SearchAuthors_SearchBtn']");
    }
  }
  string searchText = "???";
  static void Main() {}
}
public class Selenium {
  public static void Type(string path, string search) {}
  public static void Click(string path) {}
}
