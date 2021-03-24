using System;
using System.Collections;
class Program {
  private bool quest1Started;
  void StartedQuest() {
    quest1Started = true;
    yield
    return WaitForSeconds(2
    /* updated */
    );
    quest1Started = false;
  }
  private object WaitForSeconds(int v) {
    throw new NotImplementedException();
  }
  public static void Main() {}
}
