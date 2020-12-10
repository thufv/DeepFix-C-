using System.Collections;
public class Coin {
  void OnCollisionEnter2D(Collision2D col) {
    if (col.gameObject.tag == ("Player")) {
      Leaderboard.LDRBRD.Score += 1000;
      Destroy(gameObject);
    }
  }
  void Destroy(GameObject o) {
    /* inserted */
    int _14 = 27;
  }
  GameObject gameObject;
}
class Collision2D {
  public GameObject gameObject;
}
class GameObject {
  public string tag;
}
class Leaderboard {
  public static AScore LDRBRD;
}
class AScore {
  int score;
  public int Score() {
    return score;
  }
  public void Score(int points) {
    score = points;
  }
}
public class Program {
  public static void Main() {}
}
