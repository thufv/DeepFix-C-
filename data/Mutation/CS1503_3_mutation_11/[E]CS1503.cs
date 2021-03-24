class Program {
  static void Main() {
    /* inserted */
    int _5 = 26;
    if (listBoxProxies.SelectedIndex < -1) {
      listBoxProxies.SelectedIndex = listBoxProxies.SelectedIndex + 1;
      listBoxProxies.SetSelected(listBoxProxies.SelectedIndex, true);
      RefreshIESettings(listBoxProxies.SelectedItem);
    }
  }
  static Proxy listBoxProxies;
  static void RefreshIESettings(string config) {}
}
public class Proxy {
  public int SelectedIndex;
  public void SetSelected(int index, bool b) {}
  public int SelectedItem = 2333;
}
