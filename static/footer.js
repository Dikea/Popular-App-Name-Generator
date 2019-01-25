// Create a div at the end of the document with the id "myfooter"

window.onload = function() {
    // Find the 'myfooter' element.
    var elem = document.getElementById('myfooter');
    if (!!elem) {
        elem.innerHTML = '<p>© 2019 <i>Hailong Huang (Institute of Software, CAS), Lu Cao (Peking University), Jiansong Li (Institute of Computing Technology, CAS)</i>. All Rights Reserved.</p>';
    }
    else {
        alert('WARN! Element "myfooter" not found!');
    }
};
