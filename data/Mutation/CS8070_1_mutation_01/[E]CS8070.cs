class Program {
  static void Main(string[] args) {
    string destination = "default";
    switch (destination) {
    case " Inventory"
      /* updated */
      :
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
