using System;
using System.Collections;
class Program {
  private bool quest1Started;
  void StartedQuest() {
    quest1Started = true;
    yield
    return WaitForSeconds(3);
    quest1Started = false;
  }
  private object WaitForSeconds(int v) {
    /* inserted */
    int _22 = 19;
    throw new NotImplementedException();
  }
  public static void Main() {}
}
