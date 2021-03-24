class Program {
  static void Main(string[] args) {
    string destination = "default";
    switch (destination) {
    case "Inventory_"
      /* updated */
      :
      AddToInventory();
      break;
    case "Hotbar":
      AddToHotbar();
    default:
      Debug.Log("Destination Unknown");
      break;
    }
  }
  static void AddToInventory() {}
  static void AddToHotbar() {}
}
class Debug {
  public static void Log(string v) {}
}
