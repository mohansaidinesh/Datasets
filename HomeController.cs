using System.Collections.Generic;
using System.Diagnostics;
using System.Net.Http;
using System.Threading.Tasks;
using System.Xml.Linq;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Logging;
using MvcMovie.Models;

namespace MvcMovie.Controllers
{
    public class HomeController : Controller
    {
        private readonly ILogger<HomeController> _logger;

        public HomeController(ILogger<HomeController> logger)
        {
            _logger = logger;
        }

        public async Task<IActionResult> Index()
        {
            string url = "https://pro.e-box.co.in/uploads/stallxml.xml";
            XDocument xmlDoc = null;
            List<Stall> stalls = new List<Stall>();

            using (HttpClient client = new HttpClient())
            {
                string xmlContent = await client.GetStringAsync(url);
                xmlDoc = XDocument.Parse(xmlContent);

                foreach (var item in xmlDoc.Descendants("Stall"))
                {
                    Stall stall = new Stall(item.Element("Name").Value, item.Element("Type").Value);
                    stalls.Add(stall);
                }
            }
            return View(stalls);
        }

        public IActionResult Privacy()
        {
            return View();
        }

        [ResponseCache(Duration = 0, Location = ResponseCacheLocation.None, NoStore = true)]
        public IActionResult Error()
        {
            return View(new ErrorViewModel { RequestId = Activity.Current?.Id ?? HttpContext.TraceIdentifier });
        }
    }
}
