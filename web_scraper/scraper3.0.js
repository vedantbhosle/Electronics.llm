import puppeteer from "puppeteer";
import fs from "fs";

async function scrapeArduinoProjects() {
  const browser = await puppeteer.launch({ headless: true });
  const page = await browser.newPage();

  const outputFile = "projects.jsonl";
  fs.writeFileSync(outputFile, ""); // clear file before writing

  for (let i = 0; i < 2; i++) {
    const url =
      i === 0
        ? "https://circuitdigest.com/arduino-projects"
        : `https://circuitdigest.com/arduino-projects?page=${i}`;
    console.log(`Visiting listing page: ${url}`);
    await page.goto(url, { waitUntil: "domcontentloaded" });

    const projectLinks = await page.evaluate(() => {
      return Array.from(document.querySelectorAll(".views-field-title a"))
        .map((a) => a.href)
        .filter((href) =>
          href.startsWith(
            "https://circuitdigest.com/microcontroller-projects/",
          ),
        );
    });

    console.log(
      `Found ${projectLinks.length} valid project links on page ${i + 1}.`,
    );

    for (let link of projectLinks) {
      console.log(`Scraping: ${link}`);
      await page.goto(link, { waitUntil: "domcontentloaded" });

      const projectData = await page.evaluate(() => {
        const container = document.querySelector(".content");
        if (!container) return null;

        const title =
          document.querySelector("h1")?.innerText.trim() || "Untitled";
        if (!title.toLowerCase().includes("arduino")) {
          return null; // reject if title doesn't contain "arduino"
        }

        // Remove junk elements
        container
          .querySelectorAll(
            "script, style, iframe, .ad, .adsbygoogle, .advertisement, .social-share, .toc, .welcomeextra, ins",
          )
          .forEach((el) => el.remove());

        let parts = [];
        let codes = [];

        container
          .querySelectorAll("h1,h2,h3,h4,h5,h6,p,li,pre,code")
          .forEach((el) => {
            const tag = el.tagName.toLowerCase();
            const text = el.innerText.trim();
            if (!text) return;

            if (["h1", "h2", "h3", "h4"].includes(tag)) {
              parts.push(`# ${text}`); // headings as markdown
            } else if (["pre", "code"].includes(tag)) {
              codes.push(text); // collect raw code separately
            } else {
              parts.push(text);
            }
          });

        return {
          title,
          content: parts.join("\n\n"),
          codeblocks: codes,
        };
      });

      if (projectData) {
        const record = { ...projectData };
        fs.appendFileSync(outputFile, JSON.stringify(record) + "\n");
        console.log(`Saved: ${projectData.title}`);
      } else {
        console.log(`Skipped (no 'Arduino' in title): ${link}`);
      }
    }
  }

  await browser.close();
  console.log(`Scraped data saved to ${outputFile}`);
}

scrapeArduinoProjects();
