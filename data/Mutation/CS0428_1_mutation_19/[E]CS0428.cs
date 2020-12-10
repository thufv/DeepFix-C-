using System;
public class Program {
  void OnTriggerEnter(Collider coll) {
    /* inserted */
    int _17 = 19;
    if (coll.tag == "Enemy") {
      EnemyScript enemyTransform = coll.gameObject.GetComponent < EnemyScript > ;
    }
  }
  public static void Main() {}
}
internal class EnemyScript {}
public class GameObject {
  internal T GetComponent < T > () {
    throw new NotImplementedException();
  }
}
internal class Collider {
  internal string tag = null;
  internal GameObject gameObject = null;
}
