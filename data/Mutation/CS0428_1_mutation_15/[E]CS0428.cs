using System;
public class Program {
  void OnTriggerEnter(Collider coll) {
    if (coll.tag == "Enemy") {
      EnemyScript enemyTransform = coll.gameObject.GetComponent < EnemyScript > ;
    }
  }
  public static void Main() {}
}
internal class EnemyScript {}
public class GameObject {
  internal T GetComponent < T > () {
    /* inserted */
    int _10 = 17;
    throw new NotImplementedException();
  }
}
internal class Collider {
  internal string tag = null;
  internal GameObject gameObject = null;
}
