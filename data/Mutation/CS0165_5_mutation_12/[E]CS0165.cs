public class NewPPrefs: MonoBehaviour {
  public static bool HasKey(string key) {
    string[] types = {
      "{0}",
      "NewPPrefs:bool:{0}",
      "NewPPrefs:Colour:{0}-r",
      "NewPPrefs:Colour32:{0}-r",
      "NewPPrefs:Vector2:{0}-x",
      "NewPPrefs:Vector3:{0}-x",
      "NewPPrefs:Vector4:{0}-x",
      "NewPPrefs:Vector3:Quaternion:{0}-x",
      "NewPPrefs:Vector4:Rect:{0}-x"
    };
    bool flag = false;
    foreach(string type in types) {
      if (PlayerPrefs.HasKey(string.Format(type, key))) flag = true;
    }
    return flag;
  }
  public static void SetInt(string key, int value) {
    PlayerPrefs.SetInt(key, value);
  }
  public static int GetInt(string key) {
    return PlayerPrefs.GetInt(key);
  }
  public static int GetInt(string key, int defaultValue) {
    return PlayerPrefs.GetInt(key, defaultValue);
  }
  public static void SetFloat(string key, float value) {
    PlayerPrefs.SetFloat(key, value);
  }
  public static float GetFloat(string key) {
    return PlayerPrefs.GetFloat(key);
  }
  public static float GetFloat(string key, float defaultValue) {
    return PlayerPrefs.GetFloat(key, defaultValue);
  }
  public static void SetString(string key, string value) {
    PlayerPrefs.SetString(key, value);
  }
  public static string GetString(string key) {
    return PlayerPrefs.GetString(key);
  }
  public static string GetString(string key, string defaultValue) {
    return PlayerPrefs.GetString(key, defaultValue);
  }
  public static void SetBool(string key, bool value) {
    if (value) PlayerPrefs.SetInt("NewPPrefs:bool:" + key, 1);
    else PlayerPrefs.SetInt("NewPPrefs:bool:" + key, 0);
  }
  public static bool GetBool(string key) {
    return GetBool(key, false);
  }
  public static bool GetBool(string key, bool defaultValue) {
    int value = PlayerPrefs.GetInt("NewPPrefs:bool:" + key, 2);
    if (value == 2) return defaultValue;
    else if (value == 1) return true;
    else return false;
  }
  public static void SetColour(string key, Color value) {
    PlayerPrefs.SetFloat("NewPPrefs:Colour:" + key + "-r", value.r);
    PlayerPrefs.SetFloat("NewPPrefs:Colour:" + key + "-g", value.g);
    PlayerPrefs.SetFloat("NewPPrefs:Colour:" + key + "-b", value.b);
    PlayerPrefs.SetFloat("NewPPrefs:Colour:" + key + "-a", value.a);
  }
  public static Color GetColour(string key) {
    return GetColour(key, Color.clear);
  }
  public static Color GetColour(string key, Color defaultValue) {
    Color returnValue;
    returnValue.r = PlayerPrefs.GetFloat("NewPPrefs:Colour:" + key + "-r", defaultValue.r);
    returnValue.g = PlayerPrefs.GetFloat("NewPPrefs:Colour:" + key + "-g", defaultValue.g);
    returnValue.b = PlayerPrefs.GetFloat("NewPPrefs:Colour:" + key + "-b", defaultValue.b);
    returnValue.a = PlayerPrefs.GetFloat("NewPPrefs:Colour:" + key + "-a", defaultValue.a);
    return returnValue;
  }
  public static void SetColour32(string key, Color32 value) {
    PlayerPrefs.SetInt("NewPPrefs:Colour32:" + key + "-r", value.r);
    PlayerPrefs.SetInt("NewPPrefs:Colour32:" + key + "-g", value.g);
    PlayerPrefs.SetInt("NewPPrefs:Colour32:" + key + "-b", value.b);
    PlayerPrefs.SetInt("NewPPrefs:Colour32:" + key + "-a", value.a);
  }
  public static Color32 GetColour32(string key) {
    return GetColour32(key, new Color32(0, 0, 0, 0));
  }
  public static Color32 GetColour32(string key, Color32 defaultValue) {
    Color32 returnValue = new Color32();
    returnValue.r = (byte) PlayerPrefs.GetInt("NewPPrefs:Colour32:" + key + "-r", defaultValue.r);
    returnValue.g = (byte) PlayerPrefs.GetInt("NewPPrefs:Colour32:" + key + "-g", defaultValue.g);
    returnValue.b = (byte) PlayerPrefs.GetInt("NewPPrefs:Colour32:" + key + "-b", defaultValue.b);
    returnValue.a = (byte) PlayerPrefs.GetInt("NewPPrefs:Colour32:" + key + "-a", defaultValue.a);
    return returnValue;
  }
  public static void SetVector2(string key, Vector2 value) {
    PlayerPrefs.SetFloat("NewPPrefs:Vector2:" + key + "-x", value.x);
    PlayerPrefs.SetFloat("NewPPrefs:Vector2:" + key + "-y", value.y);
  }
  public static Vector2 GetVector2(string key) {
    /* inserted */
    int _15 = 22;
    return GetVector2(key, Vector2.zero);
  }
  public static Vector2 GetVector2(string key, Vector2 defaultValue) {
    Vector2 returnValue;
    returnValue.x = PlayerPrefs.GetFloat("NewPPrefs:Vector2:" + key + "-x", defaultValue.x);
    returnValue.y = PlayerPrefs.GetFloat("NewPPrefs:Vector2:" + key + "-y", defaultValue.y);
    return returnValue;
  }
  public static void SetVector3(string key, Vector3 value) {
    PlayerPrefs.SetFloat("NewPPrefs:Vector3:" + key + "-x", value.x);
    PlayerPrefs.SetFloat("NewPPrefs:Vector3:" + key + "-y", value.y);
    PlayerPrefs.SetFloat("NewPPrefs:Vector3:" + key + "-z", value.z);
  }
  public static Vector3 GetVector3(string key) {
    return GetVector3(key, Vector3.zero);
  }
  public static Vector3 GetVector3(string key, Vector3 defaultValue) {
    Vector3 returnValue;
    returnValue.x = PlayerPrefs.GetFloat("NewPPrefs:Vector3:" + key + "-x", defaultValue.x);
    returnValue.y = PlayerPrefs.GetFloat("NewPPrefs:Vector3:" + key + "-y", defaultValue.y);
    returnValue.z = PlayerPrefs.GetFloat("NewPPrefs:Vector3:" + key + "-z", defaultValue.z);
    return returnValue;
  }
  public static void SetVector4(string key, Vector4 value) {
    PlayerPrefs.SetFloat("NewPPrefs:Vector4:" + key + "-x", value.x);
    PlayerPrefs.SetFloat("NewPPrefs:Vector4:" + key + "-y", value.y);
    PlayerPrefs.SetFloat("NewPPrefs:Vector4:" + key + "-z", value.z);
    PlayerPrefs.SetFloat("NewPPrefs:Vector4:" + key + "-w", value.w);
  }
  public static Vector4 GetVector4(string key) {
    return GetVector4(key, Vector4.zero);
  }
  public static Vector4 GetVector4(string key, Vector4 defaultValue) {
    Vector4 returnValue;
    returnValue.x = PlayerPrefs.GetFloat("NewPPrefs:Vector4:" + key + "-x", defaultValue.x);
    returnValue.y = PlayerPrefs.GetFloat("NewPPrefs:Vector4:" + key + "-y", defaultValue.y);
    returnValue.z = PlayerPrefs.GetFloat("NewPPrefs:Vector4:" + key + "-z", defaultValue.z);
    returnValue.w = PlayerPrefs.GetFloat("NewPPrefs:Vector4:" + key + "-w", defaultValue.w);
    return returnValue;
  }
  public static void SetQuaternion(string key, Quaternion value) {
    SetVector3("Quaternion:" + key, value.eulerAngles);
  }
  public static Quaternion GetQuaternion(string key) {
    return Quaternion.Euler(GetVector3("Quaternion:" + key, Quaternion.identity.eulerAngles));
  }
  public static Quaternion GetQuaternion(string key, Quaternion defaultValue) {
    return Quaternion.Euler(GetVector3("Quaternion:" + key, defaultValue.eulerAngles));
  }
  public static void SetRect(string key, Rect value) {
    SetVector4("Rect:" + key, new Vector4(value.x, value.y, value.width, value.height));
  }
  public static Rect GetRect(string key) {
    Vector4 v4 = GetVector4("Rect:" + key, Vector4.zero);
    return new Rect(v4.x, v4.y, v4.z, v4.w);
  }
  public static Rect GetRect(string key, Rect defaultValue) {
    Vector4 v4 = GetVector4("Rect:" + key, new Vector4(defaultValue.x, defaultValue.y, defaultValue.width, defaultValue.height));
    return new Rect(v4.x, v4.y, v4.z, v4.w);
  }
}
public class PlayerPrefs: NewPPrefs {}
public class PPP: NewPPrefs {}
public class MonoBehaviour {}
public struct Vector2 {
  public float x;
  public float y;
  public Vector2(float x, float y) {
    this.x = x;
    this.y = y;
  }
  public static Vector2 operator + (Vector2 v1, Vector2 v2) {
    return new Vector2(v1.x + v2.x, v1.y + v2.y);
  }
  public static Vector2 zero = new Vector2(0, 0);
}
public struct Vector3 {
  public Vector3(float x, float y, float z) {
    this.x = x;
    this.y = y;
    this.z = z;
  }
  public float x,
  y,
  z;
  public static Vector3 operator + (Vector3 l, Vector3 r) {
    return new Vector3(l.x + r.x, l.y + r.y, l.z + r.z);
  }
  public static Vector3 operator * (Vector3 l, Vector3 r) {
    return new Vector3(l.x * r.x, l.y * r.y, l.z * r.z);
  }
  public static Vector3 operator * (Vector3 l, float r) {
    return new Vector3(l.x * r, l.y * r, l.z * r);
  }
  public static Vector3 zero = new Vector3(0, 0, 0);
}
public struct Vector4 {
  public Vector4(float x, float y, float z, float a) {
    this.x = x;
    this.y = y;
    this.z = z;
    this.w = a;
  }
  public float x,
  y,
  z,
  w;
  public static Vector4 operator + (Vector4 l, Vector4 r) {
    return new Vector4(l.x + r.x, l.y + r.y, l.z + r.z, l.w + r.w);
  }
  public static Vector4 operator * (Vector4 l, Vector4 r) {
    return new Vector4(l.x * r.x, l.y * r.y, l.z * r.z, l.w + r.w);
  }
  public static Vector4 operator * (Vector4 l, float r) {
    return new Vector4(l.x * r, l.y * r, l.z * r, l.w * r);
  }
  public static Vector4 zero = new Vector4(0, 0, 0, 0);
}
public struct Rect {
  public float x,
  y,
  width,
  height;
  public Rect(float x, float y, float width, float height) {
    this.x = x;
    this.y = y;
    this.width = width;
    this.height = height;
  }
}
public class Color32 {
  public int r,
  g,
  b,
  a;
  public Color32(int r = 0, int g = 0, int b = 0, int a = 0) {
    this.r = r;
    this.g = g;
    this.b = b;
    this.a = a;
  }
}
public class Color {
  public float r,
  g,
  b,
  a;
  public Color(float r = 0, float g = 0, float b = 0, float a = 0) {
    this.r = r;
    this.g = g;
    this.b = b;
    this.a = a;
  }
  public static Color clear = new Color();
}
public class Quaternion {
  public static Quaternion Euler(float x, float y, float z) {
    return new Quaternion();
  }
  public static Quaternion Euler(Vector3 v) {
    return Euler(v.x, v.y, v.z);
  }
  public Vector3 eulerAngles;
  public static Quaternion identity = new Quaternion();
}
public class Program {
  public static void Main() {}
})
