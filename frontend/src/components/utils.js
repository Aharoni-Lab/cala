const setAttributes = (el, attrs) =>
  Object.keys(attrs)
    .filter((key) => el[key] !== undefined)
    .forEach((key) =>
      typeof attrs[key] === "object"
        ? Object.keys(attrs[key]).forEach(
            (innerKey) => (el[key][innerKey] = attrs[key][innerKey]),
          )
        : (el[key] = attrs[key]),
    );

export default setAttributes;
