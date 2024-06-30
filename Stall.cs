namespace MvcMovie.Models
{
    public class Stall
    {
        private string _name;
        private string _type;

        public string Name { get { return _name; } set { _name = value; } }
        public string Type { get { return _type; } set { _type = value; } }

        public Stall() { }

        public Stall(string name, string type)
        {
            _name = name;
            _type = type;
        }
    }
}
