class Program {
  static void Main(string[] args) {
    string destination = "default";
    /* inserted */
    int _9 = 11;
    switch (destination) {
    case "Inventory":
      AddToInventory();
      break;
    case "Hotbar":
      AddToHotbar();
      break;
    default:
      Debug.Log("Destination Unknown");
    }
  }
  static void AddToInventory() {}
  static void AddToHotbar() {}
}
class Debug {
  public static void Log(string v) {}
}
