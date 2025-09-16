function getCookie(name) {
  const value = `; ${document.cookie}`;
  const parts = value.split(`; ${name}=`);
  if (parts.length === 2) return parts.pop().split(";").shift();
}

function getConfig() {
  let config_byt = getCookie("config");
  if (config_byt[0] === '"') {
    config_byt = config_byt.slice(1);
  }
  if (config_byt[config_byt.length - 1] === '"') {
    config_byt = config_byt.slice(0, -1);
  }

  const config_str = atob(config_byt);
  return JSON.parse(config_str);
}

export default getConfig;
