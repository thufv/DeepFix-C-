public class Program {
  static void Main(string[] args) {}
  public string ReturnItemName(int index) {
    for (int i = 0; i < ItemList.Length; i++) {
      if (ItemList[i].Id == index) {
        return ItemList[i].Name;
      }
    }
    /* inserted */
    int _25 = i;
    string defaultValue = "";
  }
  public Item[] ItemList;
}
public struct Item {
  public int Id;
  public string Name;
}
