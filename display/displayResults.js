function displayResults(expInfoByMethod, methodPath, methodTitle, showGIF = false) {
	document.write('<table border="1">');

	document.write("<tr>");
	// extract model name from folder name
	for(var methodI = 0; methodI < expInfoByMethod.length; methodI++) {
		document.write("<th style='color: black; text-align: center;'>" +
			methodTitle[methodI] + "</th>");
	}
	document.write("</tr>");

	// get maximum number of examples among different methods
	var exampleAmt = 0;
	for(var methodI = 0; methodI < expInfoByMethod.length; methodI++) {
		if(exampleAmt < expInfoByMethod[methodI].length) {
			exampleAmt = expInfoByMethod[methodI].length;
		}
	}

	// display each example row
	for(var exampleI = 0; exampleI < exampleAmt; exampleI++) {
		document.write("<tr>");
		// display each method column for the current example
		for(var methodI = 0; methodI < expInfoByMethod.length; methodI++) {
	  		if(exampleI < expInfoByMethod[methodI].length) {
		    	document.write("<td style='text-align: center;'>");
		    	var words = expInfoByMethod[methodI][exampleI].resultFolderName.split('_');
			    for(var wordI = 0; wordI < words.length - 5; wordI++) {
			      	document.write(words[wordI] + "_");
			    }
	    		document.write(words[words.length - 5] + "&emsp;(" + 
	      			expInfoByMethod[methodI][exampleI].vertAmt + " vertices, " + 
	      			expInfoByMethod[methodI][exampleI].faceAmt +" triangles)" + 
	      			"&emsp;<a target = '_blank' href='" + methodPath[methodI] + '/' + expInfoByMethod[methodI][exampleI].resultFolderName + "/finalResult_mesh.obj'>.obj file with UV map</a>" + "<br />");
			    if(showGIF) {
			      document.write("<a target='_blank' href = '" + methodPath[methodI] + '/' + expInfoByMethod[methodI][exampleI].resultFolderName + "/anim.gif'><img src='" + methodPath[methodI] + '/' + expInfoByMethod[methodI][exampleI].resultFolderName + "/anim.gif' width = '300'/></a>");
			    }
			    document.write("<a target='_blank' href = '" + methodPath[methodI] + '/' + expInfoByMethod[methodI][exampleI].resultFolderName + "/finalResult.png'><img src='" + methodPath[methodI] + '/' + expInfoByMethod[methodI][exampleI].resultFolderName + "/finalResult.png' width = '300'/></a>");
			    document.write("<a target='_blank' href = '" + methodPath[methodI] + '/' + expInfoByMethod[methodI][exampleI].resultFolderName + "/3DView0_distortion.png'><img src='" + methodPath[methodI] + '/' + expInfoByMethod[methodI][exampleI].resultFolderName + "/3DView0_distortion.png' width = '300'/></a>");
			    document.write("<br />" +
			    	"E<sub>d</sub> = " + parseFloat(expInfoByMethod[methodI][exampleI].E_SD).toFixed(3) + 
			    	"&emsp; E<sub>s</sub> = " + parseFloat(expInfoByMethod[methodI][exampleI].seamLen).toFixed(3) +
			    	"&emsp; time = ");
			    if(isNaN(expInfoByMethod[methodI][exampleI].time_world)) {
			    	document.write('N/A');
			    }
			    else {
			    	document.write(parseFloat(expInfoByMethod[methodI][exampleI].time_world).toFixed(3) + "s");
			    }
			    document.write("&emsp; L<sup>2</sup> stretch = " + parseFloat(expInfoByMethod[methodI][exampleI].stretch_L2).toFixed(3) +
			    	"</td>");
			}
			else {
				document.write("<td text-align: center;'></td>");
			}
		}
		document.write("</tr>");
	}

	document.write('</table>');
}
