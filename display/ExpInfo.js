// define experiment info structure
function makeStruct(names) {
  var names = names.split(' ');
  var count = names.length;
  function constructor() {
    for (var i = 0; i < count; i++) {
      this[names[i]] = arguments[i];
    }
  }
  return constructor;
}
var ExpInfo = makeStruct("vertAmt faceAmt iterAmt topoIterAmt iterAmt_rb topoIterAmt_rb lambda_init lambda_conv time time_topo time_world time_step time_step_name E_SD seamLen stretch_L2 stretch_inf shear resultFolderName");