using System;
using System.Collections.Generic;
public class Program {
  private Dictionary < string,
  int > ExtractNodeInfo(string fileContent) {
    XmlDocument xmlDocument;
    xmlDocument = new XmlDocument();
    xmlDocument.Load(fileContent);
    var ediNodes = xmlDocument.DocumentElement.SelectNodes("/EDI");
    Dictionary < string,
    int > defaultValue = new Dictionary < string,
    int > ();
    foreach(XmlNode nodes in ediNodes) {
      FileManager.nodeRecurse(nodes, defaultValue);
    }
    foreach(var entry in defaultValue) {
      Console.WriteLine(entry.ToString());
    }
  }
  static void Main() {}
}
public class XmlDocument {
  public void Load(string content) {}
  public Element DocumentElement;
}
public class Element {
  public IEnumerable < XmlNode > SelectNodes(string path) {
    /* inserted */
    int _22 = 28;
    yield
    return new XmlNode();
  }
}
public class XmlNode {}
public class FileManager {
  public static void nodeRecurse(XmlNode node, Dictionary < string, int > defaultV) {}
}
